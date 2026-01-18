import inspect
import logging
import stevedore
def get_command_names(self, group=None):
    """Returns a list of commands loaded for the specified group"""
    group_list = []
    if group is not None:
        for ep in stevedore.ExtensionManager(group):
            cmd_name = ep.name.replace('_', ' ') if self.convert_underscores else ep.name
            group_list.append(cmd_name)
        return group_list
    return list(self.commands.keys())