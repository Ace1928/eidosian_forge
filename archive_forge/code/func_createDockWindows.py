from PySide2.QtCore import QDate, QFile, Qt, QTextStream
from PySide2.QtGui import (QFont, QIcon, QKeySequence, QTextCharFormat,
from PySide2.QtPrintSupport import QPrintDialog, QPrinter
from PySide2.QtWidgets import (QAction, QApplication, QDialog, QDockWidget,
import dockwidgets_rc
def createDockWindows(self):
    dock = QDockWidget('Customers', self)
    dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
    self.customerList = QListWidget(dock)
    self.customerList.addItems(('John Doe, Harmony Enterprises, 12 Lakeside, Ambleton', 'Jane Doe, Memorabilia, 23 Watersedge, Beaton', 'Tammy Shea, Tiblanka, 38 Sea Views, Carlton', 'Tim Sheen, Caraba Gifts, 48 Ocean Way, Deal', 'Sol Harvey, Chicos Coffee, 53 New Springs, Eccleston', 'Sally Hobart, Tiroli Tea, 67 Long River, Fedula'))
    dock.setWidget(self.customerList)
    self.addDockWidget(Qt.RightDockWidgetArea, dock)
    self.viewMenu.addAction(dock.toggleViewAction())
    dock = QDockWidget('Paragraphs', self)
    self.paragraphsList = QListWidget(dock)
    self.paragraphsList.addItems(('Thank you for your payment which we have received today.', 'Your order has been dispatched and should be with you within 28 days.', 'We have dispatched those items that were in stock. The rest of your order will be dispatched once all the remaining items have arrived at our warehouse. No additional shipping charges will be made.', 'You made a small overpayment (less than $5) which we will keep on account for you, or return at your request.', "You made a small underpayment (less than $1), but we have sent your order anyway. We'll add this underpayment to your next bill.", 'Unfortunately you did not send enough money. Please remit an additional $. Your order will be dispatched as soon as the complete amount has been received.', 'You made an overpayment (more than $5). Do you wish to buy more items, or should we return the excess to you?'))
    dock.setWidget(self.paragraphsList)
    self.addDockWidget(Qt.RightDockWidgetArea, dock)
    self.viewMenu.addAction(dock.toggleViewAction())
    self.customerList.currentTextChanged.connect(self.insertCustomer)
    self.paragraphsList.currentTextChanged.connect(self.addParagraph)