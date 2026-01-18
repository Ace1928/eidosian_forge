import inspect
import logging
import stevedore
def get_command_groups(self):
    """Returns a list of the loaded command groups"""
    return self.group_list