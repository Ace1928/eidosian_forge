import warnings
def sentinel_failover(self, new_master_name):
    """
        Force a failover as if the master was not reachable, and without
        asking for agreement to other Sentinels (however a new version of the
        configuration will be published so that the other Sentinels will
        update their configurations).
        """
    return self.execute_command('SENTINEL FAILOVER', new_master_name)