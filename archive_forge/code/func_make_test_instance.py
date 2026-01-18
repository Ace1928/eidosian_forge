from .exception import MultipleMatches
from .exception import NoMatches
from .named import NamedExtensionManager
@classmethod
def make_test_instance(cls, extension, namespace='TESTING', propagate_map_exceptions=False, on_load_failure_callback=None, verify_requirements=False):
    """Construct a test DriverManager

        Test instances are passed a list of extensions to work from rather
        than loading them from entry points.

        :param extension: Pre-configured Extension instance
        :type extension: :class:`~stevedore.extension.Extension`
        :param namespace: The namespace for the manager; used only for
            identification since the extensions are passed in.
        :type namespace: str
        :param propagate_map_exceptions: Boolean controlling whether exceptions
            are propagated up through the map call or whether they are logged
            and then ignored
        :type propagate_map_exceptions: bool
        :param on_load_failure_callback: Callback function that will
            be called when an entrypoint can not be loaded. The
            arguments that will be provided when this is called (when
            an entrypoint fails to load) are (manager, entrypoint,
            exception)
        :type on_load_failure_callback: function
        :param verify_requirements: Use setuptools to enforce the
            dependencies of the plugin(s) being loaded. Defaults to False.
        :type verify_requirements: bool
        :return: The manager instance, initialized for testing

        """
    o = super(DriverManager, cls).make_test_instance([extension], namespace=namespace, propagate_map_exceptions=propagate_map_exceptions, on_load_failure_callback=on_load_failure_callback, verify_requirements=verify_requirements)
    return o