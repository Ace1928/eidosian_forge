import argparse
import importlib.util
import logging
import sys
import os
from pathlib import Path
from pprint import pprint
from typing import List, Set
from PySide6.QtCore import QCoreApplication, Qt, QLibraryInfo, QUrl, SignalInstance
from PySide6.QtGui import QGuiApplication, QSurfaceFormat
from PySide6.QtQml import QQmlApplicationEngine, QQmlComponent
from PySide6.QtQuick import QQuickView, QQuickWindow
from PySide6.QtWidgets import QApplication
def print_configurations():
    return 'Built-in configurations \n\t default \n\t resizeToItem'