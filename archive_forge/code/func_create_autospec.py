from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def create_autospec(spec, spec_set=False, instance=False, _parent=None, _name=None, **kwargs):
    """Create a mock object using another object as a spec. Attributes on the
    mock will use the corresponding attribute on the `spec` object as their
    spec.

    Functions or methods being mocked will have their arguments checked
    to check that they are called with the correct signature.

    If `spec_set` is True then attempting to set attributes that don't exist
    on the spec object will raise an `AttributeError`.

    If a class is used as a spec then the return value of the mock (the
    instance of the class) will have the same spec. You can use a class as the
    spec for an instance object by passing `instance=True`. The returned mock
    will only be callable if instances of the mock are callable.

    `create_autospec` also takes arbitrary keyword arguments that are passed to
    the constructor of the created mock."""
    if _is_list(spec):
        spec = type(spec)
    is_type = isinstance(spec, ClassTypes)
    _kwargs = {'spec': spec}
    if spec_set:
        _kwargs = {'spec_set': spec}
    elif spec is None:
        _kwargs = {}
    if _kwargs and instance:
        _kwargs['_spec_as_instance'] = True
    _kwargs.update(kwargs)
    Klass = MagicMock
    if type(spec) in DescriptorTypes:
        _kwargs = {}
    elif not _callable(spec):
        Klass = NonCallableMagicMock
    elif is_type and instance and (not _instance_callable(spec)):
        Klass = NonCallableMagicMock
    _name = _kwargs.pop('name', _name)
    _new_name = _name
    if _parent is None:
        _new_name = ''
    mock = Klass(parent=_parent, _new_parent=_parent, _new_name=_new_name, name=_name, **_kwargs)
    if isinstance(spec, FunctionTypes):
        mock = _set_signature(mock, spec)
    else:
        _check_signature(spec, mock, is_type, instance)
    if _parent is not None and (not instance):
        _parent._mock_children[_name] = mock
    if is_type and (not instance) and ('return_value' not in kwargs):
        mock.return_value = create_autospec(spec, spec_set, instance=True, _name='()', _parent=mock)
    for entry in dir(spec):
        if _is_magic(entry):
            continue
        try:
            original = getattr(spec, entry)
        except AttributeError:
            continue
        kwargs = {'spec': original}
        if spec_set:
            kwargs = {'spec_set': original}
        if not isinstance(original, FunctionTypes):
            new = _SpecState(original, spec_set, mock, entry, instance)
            mock._mock_children[entry] = new
        else:
            parent = mock
            if isinstance(spec, FunctionTypes):
                parent = mock.mock
            skipfirst = _must_skip(spec, entry, is_type)
            kwargs['_eat_self'] = skipfirst
            new = MagicMock(parent=parent, name=entry, _new_name=entry, _new_parent=parent, **kwargs)
            mock._mock_children[entry] = new
            _check_signature(original, new, skipfirst=skipfirst)
        if isinstance(new, FunctionTypes):
            setattr(mock, entry, new)
    return mock