import threading, inspect, shlex
def execCmd(self, cmdInput):
    cmdInput = cmdInput.rstrip()
    if not len(cmdInput) > 1:
        return
    if cmdInput.startswith('/'):
        cmdInput = cmdInput[1:]
    else:
        self.print_usage()
        return
    cmdInputDissect = [c for c in shlex.split(cmdInput) if c]
    cmd = cmdInputDissect[0]
    if not cmd in self.commands:
        return self.print_usage()
    cmdData = self.commands[cmd]
    if len(cmdData) == 1 and '_' in cmdData:
        subcmdData = cmdData['_']
        args = cmdInputDissect[1:] if len(cmdInputDissect) > 1 else []
    else:
        args = cmdInputDissect[2:] if len(cmdInputDissect) > 2 else []
        subcmd = cmdInputDissect[1] if len(cmdInputDissect) > 1 else ''
        if subcmd not in cmdData:
            return self.print_usage()
        subcmdData = cmdData[subcmd]
    targetFn = subcmdData['fn']
    if len(subcmdData['args']) < len(args) or len(subcmdData['args']) - subcmdData['optional'] > len(args):
        return self.print_usage()
    return self.doExecCmd(lambda: targetFn(*args))