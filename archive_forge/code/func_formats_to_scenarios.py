from breezy import repository
from breezy.bzr.remote import RemoteRepositoryFormat
from breezy.tests import default_transport, multiply_tests, test_server
from breezy.tests.per_controldir.test_controldir import TestCaseWithControlDir
from breezy.transport import memory
def formats_to_scenarios(formats, transport_server, transport_readonly_server, vfs_transport_factory=None):
    """Transform the input formats to a list of scenarios.

    :param formats: A list of (scenario_name_suffix, repo_format)
        where the scenario_name_suffix is to be appended to the format
        name, and the repo_format is a RepositoryFormat subclass
        instance.
    :returns: Scenarios of [(scenario_name, {parameter_name: value})]
    """
    result = []
    for scenario_name_suffix, repository_format in formats:
        scenario_name = repository_format.__class__.__name__
        scenario_name += scenario_name_suffix
        scenario = (scenario_name, {'transport_server': transport_server, 'transport_readonly_server': transport_readonly_server, 'bzrdir_format': repository_format._matchingcontroldir, 'repository_format': repository_format})
        if vfs_transport_factory:
            scenario[1]['vfs_transport_factory'] = vfs_transport_factory
        result.append(scenario)
    return result