import os
import sys
from PySide2.QtCore import QRect, QSize, QProcess
from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtCharts import QtCharts
def getMemoryUsage():
    result = []
    if sys.platform == 'win32':
        for line in runProcess('tasklist', [])[3:]:
            if len(line) >= 74:
                command = line[0:23].strip()
                if command.endswith('.exe'):
                    command = command[0:len(command) - 4]
                memoryUsage = float(line[64:74].strip().replace(',', '').replace('.', ''))
                legend = ''
                if memoryUsage > 10240:
                    legend = '{} {}M'.format(command, round(memoryUsage / 1024))
                else:
                    legend = '{} {}K'.format(command, round(memoryUsage))
                result.append([legend, memoryUsage])
    else:
        psOptions = ['-e', 'v']
        memoryColumn = 8
        commandColumn = 9
        if sys.platform == 'darwin':
            psOptions = ['-e', '-v']
            memoryColumn = 11
            commandColumn = 12
        for line in runProcess('ps', psOptions):
            tokens = line.split(None)
            if len(tokens) > commandColumn and 'PID' not in tokens:
                command = tokens[commandColumn]
                if not command.startswith('['):
                    command = os.path.basename(command)
                    memoryUsage = round(float(tokens[memoryColumn].replace(',', '.')))
                    legend = '{} {}%'.format(command, memoryUsage)
                    result.append([legend, memoryUsage])
    result.sort(key=lambda x: x[1], reverse=True)
    return result