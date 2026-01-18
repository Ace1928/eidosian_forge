import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def get_cpus_info(self):
    """Returns dictionary containing information about the host's CPUs."""
    cpus = self._conn_cimv2.query('SELECT Architecture, Name, Manufacturer, MaxClockSpeed, NumberOfCores, NumberOfLogicalProcessors FROM Win32_Processor WHERE ProcessorType = 3')
    cpus_list = []
    for cpu in cpus:
        cpu_info = {'Architecture': cpu.Architecture, 'Name': cpu.Name, 'Manufacturer': cpu.Manufacturer, 'MaxClockSpeed': cpu.MaxClockSpeed, 'NumberOfCores': cpu.NumberOfCores, 'NumberOfLogicalProcessors': cpu.NumberOfLogicalProcessors}
        cpus_list.append(cpu_info)
    return cpus_list