import os
import yaml
from oslo_serialization import jsonutils
from urllib import parse
from urllib import request
from mistralclient import exceptions
def do_action_on_many(action, resources, success_msg, error_msg):
    """Helper to run an action on many resources."""
    failure_flag = False
    for resource in resources:
        try:
            action(resource)
            print(success_msg % resource)
        except Exception as e:
            failure_flag = True
            print(e)
    if failure_flag:
        raise exceptions.MistralClientException(error_msg)