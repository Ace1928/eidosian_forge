from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def get_update_mask_flag():
    return base.Argument('--update-mask', help='Update mask providing the fields that are required to be updated.')