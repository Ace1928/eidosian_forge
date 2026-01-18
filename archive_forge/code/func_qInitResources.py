from PySide2 import QtCore
def qInitResources():
    QtCore.qRegisterResourceData(1, qt_resource_struct, qt_resource_name, qt_resource_data)