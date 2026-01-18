from .utils import format_environment
def inject_proxy_environment(self, environment):
    """
        Given a list of strings representing environment variables, prepend the
        environment variables corresponding to the proxy settings.
        """
    if not self:
        return environment
    proxy_env = format_environment(self.get_environment())
    if not environment:
        return proxy_env
    return proxy_env + environment