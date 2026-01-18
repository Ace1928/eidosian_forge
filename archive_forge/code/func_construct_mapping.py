from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import importlib
import os
import re
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_release_tracks
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import pkg_resources
from ruamel import yaml
import six
def construct_mapping(self, *args, **kwargs):
    data = super(Constructor, self).construct_mapping(*args, **kwargs)
    data = self._ConstructMappingHelper(Constructor.MERGE_COMMON_MACRO, self._GetCommonData, data)
    return self._ConstructMappingHelper(Constructor.MERGE_REF_MACRO, self._GetRefData, data)