from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from apitools.base.py import encoding
from googlecloudsdk.command_lib.storage.resources import full_resource_formatter
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
def is_encrypted(self):
    cmek_in_metadata = self.metadata.kmsKeyName if self.metadata else False
    return cmek_in_metadata or self.decryption_key_hash_sha256