import json
from django.contrib.messages.storage.base import BaseStorage
from django.contrib.messages.storage.cookie import MessageDecoder, MessageEncoder
from django.core.exceptions import ImproperlyConfigured
def serialize_messages(self, messages):
    encoder = MessageEncoder()
    return encoder.encode(messages)