import copy
import json
import optparse
import os
import pickle
import sys
from urllib import parse
from troveclient.compat import client
from troveclient.compat import exceptions
@classmethod
def save_from_instance_fields(cls, instance):
    apitoken = cls.default()
    for key, default_value in cls.DEFAULT_VALUES.items():
        final_value = getattr(instance, key, default_value)
        setattr(apitoken, key, final_value)
    with open(cls.APITOKEN, 'wb') as token:
        pickle.dump(apitoken, token, protocol=2)