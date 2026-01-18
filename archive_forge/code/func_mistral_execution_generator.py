import json
import uuid
from mistralclient.api.client import client as mistral_client
from troveclient import base
from troveclient import common
def mistral_execution_generator():
    m = marker
    while True:
        try:
            the_list = mistral_client.executions.list(marker=m, limit=50, sort_dirs='desc')
            if the_list:
                for the_item in the_list:
                    yield the_item
                m = the_list[-1].id
            else:
                return
        except StopIteration:
            return