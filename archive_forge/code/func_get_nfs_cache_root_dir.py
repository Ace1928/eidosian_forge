import os
import shutil
import uuid
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.databricks_utils import _get_dbutils, is_in_databricks_runtime
def get_nfs_cache_root_dir():
    if is_in_databricks_runtime():
        spark_sess = _get_active_spark_session()
        nfs_enabled = spark_sess and spark_sess.conf.get('spark.databricks.mlflow.nfs.enabled', 'true').lower() == 'true'
        if nfs_enabled:
            try:
                return _get_dbutils().entry_point.getReplNFSTempDir()
            except Exception:
                nfs_root_dir = '/local_disk0/.ephemeral_nfs'
                test_path = os.path.join(nfs_root_dir, uuid.uuid4().hex)
                try:
                    os.makedirs(test_path)
                    return nfs_root_dir
                except Exception:
                    return None
                finally:
                    shutil.rmtree(test_path, ignore_errors=True)
        else:
            return None
    else:
        spark_session = _get_active_spark_session()
        if spark_session is not None:
            return spark_session.conf.get('spark.mlflow.nfs.rootDir', None)