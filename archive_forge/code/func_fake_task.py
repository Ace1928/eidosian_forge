from unittest import mock
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware.tests import base
def fake_task(fault_class_name, error_msg=None):
    task_info = mock.Mock()
    task_info.localizedMessage = error_msg
    if fault_class_name:
        error_fault = mock.Mock()
        error_fault.__class__.__name__ = fault_class_name
        task_info.fault = error_fault
    return task_info