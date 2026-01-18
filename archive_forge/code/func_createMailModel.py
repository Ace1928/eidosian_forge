from PySide2 import QtCore, QtGui, QtWidgets
def createMailModel(parent):
    model = QtGui.QStandardItemModel(0, 3, parent)
    model.setHeaderData(0, QtCore.Qt.Horizontal, 'Subject')
    model.setHeaderData(1, QtCore.Qt.Horizontal, 'Sender')
    model.setHeaderData(2, QtCore.Qt.Horizontal, 'Date')
    addMail(model, 'Happy New Year!', 'Grace K. <grace@software-inc.com>', QtCore.QDateTime(QtCore.QDate(2006, 12, 31), QtCore.QTime(17, 3)))
    addMail(model, 'Radically new concept', 'Grace K. <grace@software-inc.com>', QtCore.QDateTime(QtCore.QDate(2006, 12, 22), QtCore.QTime(9, 44)))
    addMail(model, 'Accounts', 'pascale@nospam.com', QtCore.QDateTime(QtCore.QDate(2006, 12, 31), QtCore.QTime(12, 50)))
    addMail(model, 'Expenses', 'Joe Bloggs <joe@bloggs.com>', QtCore.QDateTime(QtCore.QDate(2006, 12, 25), QtCore.QTime(11, 39)))
    addMail(model, 'Re: Expenses', 'Andy <andy@nospam.com>', QtCore.QDateTime(QtCore.QDate(2007, 1, 2), QtCore.QTime(16, 5)))
    addMail(model, 'Re: Accounts', 'Joe Bloggs <joe@bloggs.com>', QtCore.QDateTime(QtCore.QDate(2007, 1, 3), QtCore.QTime(14, 18)))
    addMail(model, 'Re: Accounts', 'Andy <andy@nospam.com>', QtCore.QDateTime(QtCore.QDate(2007, 1, 3), QtCore.QTime(14, 26)))
    addMail(model, 'Sports', 'Linda Smith <linda.smith@nospam.com>', QtCore.QDateTime(QtCore.QDate(2007, 1, 5), QtCore.QTime(11, 33)))
    addMail(model, 'AW: Sports', 'Rolf Newschweinstein <rolfn@nospam.com>', QtCore.QDateTime(QtCore.QDate(2007, 1, 5), QtCore.QTime(12, 0)))
    addMail(model, 'RE: Sports', 'Petra Schmidt <petras@nospam.com>', QtCore.QDateTime(QtCore.QDate(2007, 1, 5), QtCore.QTime(12, 1)))
    return model