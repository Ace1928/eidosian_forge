from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.mps.mps_client import MpsClient
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.mps import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_projector
def synthesizesInstanceInfo(self, ins):
    out = resource_projector.MakeSerializable(ins)
    out['osImage'] = ins.osImage.version
    out['osImage'] = json.dumps(out['osImage'])
    return out