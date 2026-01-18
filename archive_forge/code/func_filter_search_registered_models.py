import functools
import importlib
import logging
import re
import uuid
from typing import Any, Callable, Dict, Optional, Union
import sqlalchemy
from flask import Flask, Response, flash, jsonify, make_response, render_template_string, request
from werkzeug.datastructures import Authorization
from mlflow import MlflowException
from mlflow.entities import Experiment
from mlflow.entities.model_registry import RegisteredModel
from mlflow.protos.databricks_pb2 import (
from mlflow.protos.model_registry_pb2 import (
from mlflow.protos.service_pb2 import (
from mlflow.server import app
from mlflow.server.auth.config import read_auth_config
from mlflow.server.auth.logo import MLFLOW_LOGO
from mlflow.server.auth.permissions import MANAGE, Permission, get_permission
from mlflow.server.auth.routes import (
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.server.handlers import (
from mlflow.store.entities import PagedList
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.rest_utils import _REST_API_PATH_PREFIX
from mlflow.utils.search_utils import SearchUtils
def filter_search_registered_models(resp: Response):
    if sender_is_admin():
        return
    response_message = SearchRegisteredModels.Response()
    parse_dict(resp.json, response_message)
    username = authenticate_request().username
    perms = store.list_registered_model_permissions(username)
    can_read = {p.name: get_permission(p.permission).can_read for p in perms}
    default_can_read = get_permission(auth_config.default_permission).can_read
    for rm in list(response_message.registered_models):
        if not can_read.get(rm.name, default_can_read):
            response_message.registered_models.remove(rm)
    request_message = _get_request_message(SearchRegisteredModels())
    while len(response_message.registered_models) < request_message.max_results and response_message.next_page_token != '':
        refetched: PagedList[RegisteredModel] = _get_model_registry_store().search_registered_models(filter_string=request_message.filter, max_results=request_message.max_results, order_by=request_message.order_by, page_token=response_message.next_page_token)
        refetched = refetched[:request_message.max_results - len(response_message.registered_models)]
        if len(refetched) == 0:
            response_message.next_page_token = ''
            break
        refetched_readable_proto = [rm.to_proto() for rm in refetched if can_read.get(rm.name, default_can_read)]
        response_message.registered_models.extend(refetched_readable_proto)
        start_offset = SearchUtils.parse_start_offset_from_page_token(response_message.next_page_token)
        final_offset = start_offset + len(refetched)
        response_message.next_page_token = SearchUtils.create_page_token(final_offset)
    resp.data = message_to_json(response_message)