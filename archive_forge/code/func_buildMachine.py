from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
def buildMachine(self):
    machine = QStateMachine(self)
    inputState = Custom(machine, self)
    self.status = 'hello!'
    inputState.assignProperty(self, 'status', 'Move the rogue with 2, 4, 6, and 8')
    machine.setInitialState(inputState)
    machine.start()
    transition = MovementTransition(self)
    inputState.addTransition(transition)
    quitState = QState(machine)
    quitState.assignProperty(self, 'status', 'Really quit(y/n)?')
    yesTransition = QKeyEventTransition(self, QEvent.KeyPress, Qt.Key_Y)
    self.finalState = QFinalState(machine)
    yesTransition.setTargetState(self.finalState)
    quitState.addTransition(yesTransition)
    noTransition = QKeyEventTransition(self, QEvent.KeyPress, Qt.Key_N)
    noTransition.setTargetState(inputState)
    quitState.addTransition(noTransition)
    quitTransition = QKeyEventTransition(self, QEvent.KeyPress, Qt.Key_Q)
    quitTransition.setTargetState(quitState)
    inputState.addTransition(quitTransition)
    machine.setInitialState(inputState)
    machine.finished.connect(qApp.quit)
    machine.start()