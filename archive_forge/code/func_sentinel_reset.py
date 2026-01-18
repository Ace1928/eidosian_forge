import warnings
def sentinel_reset(self, pattern):
    """
        This command will reset all the masters with matching name.
        The pattern argument is a glob-style pattern.

        The reset process clears any previous state in a master (including a
        failover in progress), and removes every slave and sentinel already
        discovered and associated with the master.
        """
    return self.execute_command('SENTINEL RESET', pattern, once=True)