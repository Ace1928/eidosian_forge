from __future__ import print_function, absolute_import
from PySide2.QtWidgets import (QAction, QAbstractItemView, qApp, QDataWidgetMapper,
from PySide2.QtGui import QKeySequence
from PySide2.QtSql import (QSqlRelation, QSqlRelationalTableModel, QSqlTableModel,
from PySide2.QtCore import QAbstractItemModel, QObject, QSize, Qt, Slot
import createdb
from ui_bookwindow import Ui_BookWindow
from bookdelegate import BookDelegate
def create_menubar(self):
    file_menu = self.menuBar().addMenu(self.tr('&File'))
    quit_action = file_menu.addAction(self.tr('&Quit'))
    quit_action.triggered.connect(qApp.quit)
    help_menu = self.menuBar().addMenu(self.tr('&Help'))
    about_action = help_menu.addAction(self.tr('&About'))
    about_action.setShortcut(QKeySequence.HelpContents)
    about_action.triggered.connect(self.about)
    aboutQt_action = help_menu.addAction('&About Qt')
    aboutQt_action.triggered.connect(qApp.aboutQt)