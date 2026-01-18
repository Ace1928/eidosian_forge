import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def describe_db_engine_versions(self, engine=None, engine_version=None, db_parameter_group_family=None, max_records=None, marker=None, default_only=None, list_supported_character_sets=None):
    """
        Returns a list of the available DB engines.

        :type engine: string
        :param engine: The database engine to return.

        :type engine_version: string
        :param engine_version: The database engine version to return.
        Example: `5.1.49`

        :type db_parameter_group_family: string
        :param db_parameter_group_family:
        The name of a specific DB parameter group family to return details for.

        Constraints:


        + Must be 1 to 255 alphanumeric characters
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

        :type max_records: integer
        :param max_records: The maximum number of records to include in the
            response. If more than the `MaxRecords` value is available, a
            pagination token called a marker is included in the response so
            that the following results can be retrieved.
        Default: 100

        Constraints: minimum 20, maximum 100

        :type marker: string
        :param marker: An optional pagination token provided by a previous
            request. If this parameter is specified, the response includes only
            records beyond the marker, up to the value specified by
            `MaxRecords`.

        :type default_only: boolean
        :param default_only: Indicates that only the default version of the
            specified engine or engine and major version combination is
            returned.

        :type list_supported_character_sets: boolean
        :param list_supported_character_sets: If this parameter is specified,
            and if the requested engine supports the CharacterSetName parameter
            for CreateDBInstance, the response includes a list of supported
            character sets for each engine version.

        """
    params = {}
    if engine is not None:
        params['Engine'] = engine
    if engine_version is not None:
        params['EngineVersion'] = engine_version
    if db_parameter_group_family is not None:
        params['DBParameterGroupFamily'] = db_parameter_group_family
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    if default_only is not None:
        params['DefaultOnly'] = str(default_only).lower()
    if list_supported_character_sets is not None:
        params['ListSupportedCharacterSets'] = str(list_supported_character_sets).lower()
    return self._make_request(action='DescribeDBEngineVersions', verb='POST', path='/', params=params)