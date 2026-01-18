import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
@classmethod
def as_user(self, user):
    clients = self.get_clients()
    if user in clients:
        return clients[user]
    raise Exception(f"User '{user}' does not exist")