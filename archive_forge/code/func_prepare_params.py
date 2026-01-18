from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
def prepare_params(self, input_dict):
    """
        Transform a dict with data into one that can be accepted as params for calling the action.

        This will ignore any keys that are not accepted as params when calling the action.
        It also allows generating nested params without forcing the user to care about them.

        :param input_dict: a dict with data that should be used to fill in the params
        :return: :class:`dict` object
        :rtype: dict

        Usage::

            >>> action.prepare_params({'id': 1})
            {'user': {'id': 1}}
        """
    params = self._prepare_params(self.params, input_dict)
    route_params = self._prepare_route_params(input_dict)
    params.update(route_params)
    return params