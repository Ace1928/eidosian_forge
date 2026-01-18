import datetime
import os
import shutil
import tempfile
import time
import types
import warnings
from dulwich.tests import SkipTest
from ..index import commit_tree
from ..objects import Commit, FixedSha, Tag, object_class
from ..pack import (
from ..repo import Repo
def make_tag(target, **attrs):
    """Make a Tag object with a default set of values.

    Args:
      target: object to be tagged (Commit, Blob, Tree, etc)
      attrs: dict of attributes to overwrite from the default values.
    Returns: A newly initialized Tag object.
    """
    target_id = target.id
    target_type = object_class(target.type_name)
    default_time = int(time.mktime(datetime.datetime(2010, 1, 1).timetuple()))
    all_attrs = {'tagger': b'Test Author <test@nodomain.com>', 'tag_time': default_time, 'tag_timezone': 0, 'message': b'Test message.', 'object': (target_type, target_id), 'name': b'Test Tag'}
    all_attrs.update(attrs)
    return make_object(Tag, **all_attrs)