import unittest
import uuid
import datetime
from boto.mturk.question import (
from ._init_environment import SetHostMTurkConnection, config_environment
@staticmethod
def get_hit_params():
    return dict(lifetime=datetime.timedelta(minutes=65), max_assignments=2, title='Boto create_hit title', description='Boto create_hit description', keywords=['boto', 'test'], reward=0.23, duration=datetime.timedelta(minutes=6), approval_delay=60 * 60, annotation='An annotation from boto create_hit test', response_groups=['Minimal', 'HITDetail', 'HITQuestion', 'HITAssignmentSummary'])