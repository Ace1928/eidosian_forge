import os
import sys
from tempfile import mkdtemp
from shutil import rmtree
import pytest
import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function
def run_multiproc_nondaemon_with_flag(nondaemon_flag):
    """
    Start a pipe with two nodes using the resource multiproc plugin and
    passing the nondaemon_flag.
    """
    cur_dir = os.getcwd()
    temp_dir = mkdtemp(prefix='test_engine_')
    os.chdir(temp_dir)
    pipe = pe.Workflow(name='pipe')
    f1 = pe.Node(interface=Function(function=mytestFunction, input_names=['insum'], output_names=['sum_out']), name='f1')
    f2 = pe.Node(interface=Function(function=mytestFunction, input_names=['insum'], output_names=['sum_out']), name='f2')
    pipe.connect([(f1, f2, [('sum_out', 'insum')])])
    pipe.base_dir = os.getcwd()
    f1.inputs.insum = 0
    pipe.config['execution']['stop_on_first_crash'] = True
    execgraph = pipe.run(plugin='LegacyMultiProc', plugin_args={'n_procs': 2, 'non_daemon': nondaemon_flag})
    names = ['.'.join((node._hierarchy, node.name)) for node in execgraph.nodes()]
    node = list(execgraph.nodes())[names.index('pipe.f2')]
    result = node.get_output('sum_out')
    os.chdir(cur_dir)
    rmtree(temp_dir)
    return result