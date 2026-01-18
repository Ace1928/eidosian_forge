from pprint import pformat
from six import iteritems
import re
@sub_path_expr.setter
def sub_path_expr(self, sub_path_expr):
    """
        Sets the sub_path_expr of this V1VolumeMount.
        Expanded path within the volume from which the container's volume should
        be mounted. Behaves similarly to SubPath but environment variable
        references $(VAR_NAME) are expanded using the container's environment.
        Defaults to "" (volume's root). SubPathExpr and SubPath are mutually
        exclusive. This field is alpha in 1.14.

        :param sub_path_expr: The sub_path_expr of this V1VolumeMount.
        :type: str
        """
    self._sub_path_expr = sub_path_expr