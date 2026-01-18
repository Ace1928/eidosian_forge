import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def disableSorting(self, w):
    if self.item_nr == 0:
        self.sorting_enabled = self.factory.invoke('__sortingEnabled', w.isSortingEnabled)
        w.setSortingEnabled(False)