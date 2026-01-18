import pytest
import functools
@functools.lru_cache(maxsize=1)
def get_relative_type_widget_classes():
    from kivy.factory import Factory
    return tuple((Factory.get(cls_name) for cls_name in relative_type_widget_cls_names))