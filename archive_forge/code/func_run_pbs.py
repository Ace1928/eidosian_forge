import os
import time
import subprocess
import re
import warnings
import numpy as np
from ase.geometry import cell_to_cellpar
from ase.calculators.calculator import (FileIOCalculator, Calculator, equal,
from ase.calculators.openmx.parameters import OpenMXParameters
from ase.calculators.openmx.default_settings import default_dictionary
from ase.calculators.openmx.reader import read_openmx, get_file_name
from ase.calculators.openmx.writer import write_openmx
def run_pbs(self, prefix='test'):
    """
        Execute the OpenMX using Plane Batch System. In order to use this,
        Your system should have Scheduler. PBS
        Basically, it does qsub. and wait until qstat signal shows c
        Super computer user
        """
    nodes = self.nodes
    processes = self.processes
    prefix = self.prefix
    olddir = os.getcwd()
    try:
        os.chdir(self.abs_directory)
    except AttributeError:
        os.chdir(self.directory)

    def isRunning(jobNum=None, status='Q', qstat='qstat'):
        """
            Check submitted job is still Running
            """

        def runCmd(exe):
            p = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            while True:
                line = p.stdout.readline()
                if line != '':
                    yield line.rstrip()
                else:
                    break
        jobs = runCmd('qstat')
        columns = None
        for line in jobs:
            if str(jobNum) in line:
                columns = line.split()
                self.prind(line)
        if columns is not None:
            return columns[-2] == status
        else:
            return False
    inputfile = self.label + '.dat'
    outfile = self.label + '.log'
    bashArgs = '#!/bin/bash \n cd $PBS_O_WORKDIR\n'
    jobName = prefix
    cmd = bashArgs + 'mpirun -hostfile $PBS_NODEFILE openmx %s > %s' % (inputfile, outfile)
    echoArgs = ['echo', "$' %s'" % cmd]
    qsubArgs = ['qsub', '-N', jobName, '-l', 'nodes=%d:ppn=%d' % (nodes, processes), '-l', 'walltime=' + self.walltime]
    wholeCmd = ' '.join(echoArgs) + ' | ' + ' '.join(qsubArgs)
    self.prind(wholeCmd)
    out = subprocess.Popen(wholeCmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    out = out.communicate()[0]
    jobNum = int(re.match('(\\d+)', out.split()[0]).group(1))
    self.prind('Queue number is ' + str(jobNum) + '\nWaiting for the Queue to start')
    while isRunning(jobNum, status='Q'):
        time.sleep(5)
        self.prind('.')
    self.prind('Start Calculating')
    self.print_file(file=outfile, running=isRunning, jobNum=jobNum, status='R', qstat='qstat')
    os.chdir(olddir)
    self.prind('Calculation Finished!')
    return jobNum