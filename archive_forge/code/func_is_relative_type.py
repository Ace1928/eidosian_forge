import pytest
import functools
def is_relative_type(widget):
    return isinstance(widget, get_relative_type_widget_classes())