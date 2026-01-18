from suds import *
from suds.umx import *
from suds.umx.core import Core
from suds.resolver import NodeResolver, Frame
from suds.sudsobject import Factory
from logging import getLogger
def multi_occurrence(self, content):
    return content.type.multi_occurrence()