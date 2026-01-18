from ipywidgets import Widget
import ipywidgets.widgets.widget
import ipykernel.comm
def teardown_test_comm():
    if NEW_COMM_PACKAGE:
        comm.create_comm = orig_create_comm
        comm.get_comm_manager = orig_get_comm_manager
        ipykernel.comm.comm.BaseComm = orig_comm
    else:
        ipykernel.comm.Comm = orig_comm
    Widget.comm.klass = orig_comm
    ipywidgets.widgets.widget.Comm = orig_comm
    for attr, value in _widget_attrs.items():
        if value is undefined:
            delattr(Widget, attr)
        else:
            setattr(Widget, attr, value)
    _widget_attrs.clear()