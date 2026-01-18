import dataclasses
import json
import logging
import socket
import sys
import threading
import traceback
import urllib.parse
from collections import defaultdict, deque
from copy import deepcopy
from typing import (
import flask
import pandas as pd
import requests
import responses
import wandb
import wandb.util
from wandb.sdk.lib.timer import Timer
def relay(self, request: 'flask.Request') -> Union['responses.Response', 'requests.Response']:
    url = urllib.parse.urlparse(request.url)._replace(netloc=self.base_url.netloc, scheme=self.base_url.scheme).geturl()
    headers = {key: value for key, value in request.headers if key != 'Host'}
    prepared_relayed_request = requests.Request(method=request.method, url=url, headers=headers, data=request.get_data(), json=request.get_json()).prepare()
    if self.verbose:
        print('*****************')
        print('RELAY REQUEST:')
        print(prepared_relayed_request.url)
        print(prepared_relayed_request.method)
        print(prepared_relayed_request.headers)
        print(prepared_relayed_request.body)
        print('*****************')
    for injected_response in self.inject:
        should_apply = injected_response.application_pattern.should_apply()
        if injected_response == prepared_relayed_request:
            if self.verbose:
                print('*****************')
                print('INJECTING RESPONSE:')
                print(injected_response.to_dict())
                print('*****************')
            injected_response.application_pattern.next()
            if should_apply:
                with responses.RequestsMock() as mocked_responses:
                    mocked_responses.add(**injected_response.to_dict())
                    relayed_response = self.session.send(prepared_relayed_request)
                    return relayed_response
    relayed_response = self.session.send(prepared_relayed_request)
    return relayed_response