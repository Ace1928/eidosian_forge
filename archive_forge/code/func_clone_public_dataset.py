from __future__ import annotations
import collections
import datetime
import functools
import importlib
import importlib.metadata
import io
import json
import logging
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue, Queue
from typing import (
from urllib import parse as urllib_parse
import orjson
import requests
from requests import adapters as requests_adapters
from urllib3.util import Retry
import langsmith
from langsmith import env as ls_env
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def clone_public_dataset(self, token_or_url: str, *, source_api_url: Optional[str]=None, dataset_name: Optional[str]=None) -> None:
    """Clone a public dataset to your own langsmith tenant.

        This operation is idempotent. If you already have a dataset with the given name,
        this function will do nothing.

        Args:
            token_or_url (str): The token of the public dataset to clone.
            source_api_url: The URL of the langsmith server where the data is hosted.
                Defaults to the API URL of your current client.
            dataset_name (str): The name of the dataset to create in your tenant.
                Defaults to the name of the public dataset.

        """
    source_api_url = source_api_url or self.api_url
    source_api_url, token_uuid = _parse_token_or_url(token_or_url, source_api_url)
    source_client = Client(api_url=source_api_url, api_key='placeholder')
    ds = source_client.read_shared_dataset(token_uuid)
    dataset_name = dataset_name or ds.name
    if self.has_dataset(dataset_name=dataset_name):
        logger.info(f'Dataset {dataset_name} already exists in your tenant. Skipping.')
        return
    try:
        examples = list(source_client.list_shared_examples(token_uuid))
        dataset = self.create_dataset(dataset_name=dataset_name, description=ds.description, data_type=ds.data_type or ls_schemas.DataType.kv)
        try:
            self.create_examples(inputs=[e.inputs for e in examples], outputs=[e.outputs for e in examples], dataset_id=dataset.id)
        except BaseException as e:
            logger.error(f'An error occurred while creating dataset {dataset_name}. You should delete it manually.')
            raise e
    finally:
        del source_client