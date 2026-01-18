from ipywidgets import Widget
import ipywidgets.widgets.widget
import ipykernel.comm
def dummy_create_comm(**kwargs):
    return DummyComm()