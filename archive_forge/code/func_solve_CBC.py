from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .core import cbc_path, pulp_cbc_path, coinMP_path, devnull, operating_system
import os
from .. import constants
from tempfile import mktemp
import ctypes
import warnings
def solve_CBC(self, lp, use_mps=True):
    """Solve a MIP problem using CBC"""
    if not self.executable(self.path):
        raise PulpSolverError(f'Pulp: cannot execute {self.path} cwd: {os.getcwd()}')
    tmpLp, tmpMps, tmpSol, tmpMst = self.create_tmp_files(lp.name, 'lp', 'mps', 'sol', 'mst')
    if use_mps:
        vs, variablesNames, constraintsNames, objectiveName = lp.writeMPS(tmpMps, rename=1)
        cmds = ' ' + tmpMps + ' '
        if lp.sense == constants.LpMaximize:
            cmds += '-max '
    else:
        vs = lp.writeLP(tmpLp)
        variablesNames = {v.name: v.name for v in vs}
        constraintsNames = {c: c for c in lp.constraints}
        cmds = ' ' + tmpLp + ' '
    if self.optionsDict.get('warmStart', False):
        self.writesol(tmpMst, lp, vs, variablesNames, constraintsNames)
        cmds += f'-mips {tmpMst} '
    if self.timeLimit is not None:
        cmds += f'-sec {self.timeLimit} '
    options = self.options + self.getOptions()
    for option in options:
        cmds += '-' + option + ' '
    if self.mip:
        cmds += '-branch '
    else:
        cmds += '-initialSolve '
    cmds += '-printingOptions all '
    cmds += '-solution ' + tmpSol + ' '
    if self.msg:
        pipe = None
    else:
        pipe = open(os.devnull, 'w')
    logPath = self.optionsDict.get('logPath')
    if logPath:
        if self.msg:
            warnings.warn('`logPath` argument replaces `msg=1`. The output will be redirected to the log file.')
        pipe = open(self.optionsDict['logPath'], 'w')
    log.debug(self.path + cmds)
    args = []
    args.append(self.path)
    args.extend(cmds[1:].split())
    if not self.msg and operating_system == 'win':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        cbc = subprocess.Popen(args, stdout=pipe, stderr=pipe, stdin=devnull, startupinfo=startupinfo)
    else:
        cbc = subprocess.Popen(args, stdout=pipe, stderr=pipe, stdin=devnull)
    if cbc.wait() != 0:
        if pipe:
            pipe.close()
        raise PulpSolverError('Pulp: Error while trying to execute, use msg=True for more details' + self.path)
    if pipe:
        pipe.close()
    if not os.path.exists(tmpSol):
        raise PulpSolverError('Pulp: Error while executing ' + self.path)
    status, values, reducedCosts, shadowPrices, slacks, sol_status = self.readsol_MPS(tmpSol, lp, vs, variablesNames, constraintsNames)
    lp.assignVarsVals(values)
    lp.assignVarsDj(reducedCosts)
    lp.assignConsPi(shadowPrices)
    lp.assignConsSlack(slacks, activity=True)
    lp.assignStatus(status, sol_status)
    self.delete_tmp_files(tmpMps, tmpLp, tmpSol, tmpMst)
    return status