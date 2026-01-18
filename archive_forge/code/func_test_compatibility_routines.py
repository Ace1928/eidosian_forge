from collections import namedtuple
from collections import OrderedDict
import copy
import datetime
import inspect
import logging
from unittest import mock
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import versionutils as vutils
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
def test_compatibility_routines(self, use_manifest=False, init_args=None, init_kwargs=None):
    """Test obj_make_compatible() on all object classes.

        :param use_manifest: a boolean that determines if the version
                             manifest should be passed to obj_make_compatible
        :param init_args: a dictionary of the format {obj_class: [arg1, arg2]}
                          that will be used to pass arguments to init on the
                          given obj_class. If no args are needed, the
                          obj_class does not need to be added to the dict
        :param init_kwargs: a dictionary of the format
                            {obj_class: {'kwarg1': val1}} that will be used to
                            pass kwargs to init on the given obj_class. If no
                            kwargs are needed, the obj_class does not need to
                            be added to the dict
        """
    init_args = init_args or {}
    init_kwargs = init_kwargs or {}
    for obj_name in self.obj_classes:
        obj_classes = self.obj_classes[obj_name]
        if use_manifest:
            manifest = base.obj_tree_get_versions(obj_name)
        else:
            manifest = None
        for obj_class in obj_classes:
            args_for_init = init_args.get(obj_class, [])
            kwargs_for_init = init_kwargs.get(obj_class, {})
            self._test_object_compatibility(obj_class, manifest=manifest, init_args=args_for_init, init_kwargs=kwargs_for_init)