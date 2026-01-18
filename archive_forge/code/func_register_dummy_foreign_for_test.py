from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
def register_dummy_foreign_for_test(testcase):
    controldir.ControlDirFormat.register_prober(DummyForeignProber)
    testcase.addCleanup(controldir.ControlDirFormat.unregister_prober, DummyForeignProber)
    repository.format_registry.register(DummyForeignVcsRepositoryFormat())
    testcase.addCleanup(repository.format_registry.remove, DummyForeignVcsRepositoryFormat())
    branch.format_registry.register(DummyForeignVcsBranchFormat())
    testcase.addCleanup(branch.format_registry.remove, DummyForeignVcsBranchFormat())
    branch.InterBranch.register_optimiser(InterToDummyVcsBranch)
    testcase.addCleanup(branch.InterBranch.unregister_optimiser, InterToDummyVcsBranch)