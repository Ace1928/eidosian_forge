from nipype.pipeline.plugins.base import SGELikeBatchManagerBase
from nipype.interfaces.utility import Function
import nipype.pipeline.engine as pe
import pytest
from unittest.mock import patch
import subprocess
@patch.object(SGELikeBatchManagerBase, '_submit_batchtask', new=submit_batchtask)
@patch.object(SGELikeBatchManagerBase, '_is_pending', new=is_pending)
def test_crashfile_creation(tmp_path):
    pipe = pe.Workflow(name='pipe', base_dir=str(tmp_path))
    pipe.config['execution']['crashdump_dir'] = str(tmp_path)
    pipe.add_nodes([pe.Node(interface=Function(function=crasher), name='crasher')])
    sgelike_plugin = SGELikeBatchManagerBase('')
    with pytest.raises(RuntimeError) as e:
        assert pipe.run(plugin=sgelike_plugin)
    crashfiles = list(tmp_path.glob('crash*crasher*.pklz')) + list(tmp_path.glob('crash*crasher*.txt'))
    assert len(crashfiles) == 1