import warnings
def sentinel_ckquorum(self, new_master_name):
    """
        Check if the current Sentinel configuration is able to reach the
        quorum needed to failover a master, and the majority needed to
        authorize the failover.

        This command should be used in monitoring systems to check if a
        Sentinel deployment is ok.
        """
    return self.execute_command('SENTINEL CKQUORUM', new_master_name, once=True)