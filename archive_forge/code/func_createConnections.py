import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def createConnections(self, elem):

    def name2object(obj):
        if obj == self.uiname:
            return self.toplevelWidget
        else:
            return getattr(self.toplevelWidget, obj)
    for conn in iter(elem):
        signal = conn.findtext('signal')
        signal_name, signal_args = signal.split('(')
        signal_args = signal_args[:-1].replace(' ', '')
        sender = name2object(conn.findtext('sender'))
        bound_signal = getattr(sender, signal_name)
        slot = self.factory.getSlot(name2object(conn.findtext('receiver')), conn.findtext('slot').split('(')[0])
        if signal_args == '':
            bound_signal.connect(slot)
        else:
            signal_args = signal_args.split(',')
            if len(signal_args) == 1:
                bound_signal[signal_args[0]].connect(slot)
            else:
                bound_signal[tuple(signal_args)].connect(slot)
    QtCore.QMetaObject.connectSlotsByName(self.toplevelWidget)