from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def report_detail_attribute_config():
    return concepts.ResourceParameterAttributeConfig(name='report-detail', help_text='Report Detail ID for the {resource}.')