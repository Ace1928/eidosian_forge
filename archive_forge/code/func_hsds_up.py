import os
import sys
import tempfile
from pathlib import Path
from shutil import rmtree
import pytest
@pytest.fixture(scope='session')
def hsds_up():
    """Provide HDF Highly Scalabale Data Service (HSDS) for h5pyd testing."""
    if with_reqd_pkgs:
        root_dir = Path(tempfile.mkdtemp(prefix='tmp-hsds-root-'))
        os.environ['BUCKET_NAME'] = 'data'
        (root_dir / os.getenv('BUCKET_NAME')).mkdir(parents=True, exist_ok=True)
        os.environ['ROOT_DIR'] = str(root_dir)
        os.environ['HS_USERNAME'] = 'h5netcdf-pytest'
        os.environ['HS_PASSWORD'] = 'TestEarlyTestEverything'
        config = 'allow_noauth: true\nauth_expiration: -1\ndefault_public: False\naws_access_key_id: xxx\naws_secret_access_key: xxx\naws_iam_role: hsds_role\naws_region: us-east-1\nhsds_endpoint: http://hsds.hdf.test\naws_s3_gateway: null\naws_dynamodb_gateway: null\naws_dynamodb_users_table: null\nazure_connection_string: null\nazure_resource_id: null\nazure_storage_account: null\nazure_resource_group: null\nroot_dir: null\npassword_salt: null\nbucket_name: hsdstest\nhead_port: 5100\nhead_ram: 512m\ndn_port: 6101\ndn_ram: 3g\nsn_port: 5101\nsn_ram: 1g\nrangeget_port: 6900\nrangeget_ram: 2g\ntarget_sn_count: 0\ntarget_dn_count: 0\nlog_level: INFO\nlog_timestamps: false\nlog_prefix: null\nmax_tcp_connections: 100\nhead_sleep_time: 10\nnode_sleep_time: 10\nasync_sleep_time: 10\ns3_sync_interval: 1\ns3_sync_task_timeout: 10\nstore_read_timeout: 1\nstore_read_sleep_interval: 0.1\nmax_pending_write_requests: 20\nflush_sleep_interval: 1\nmax_chunks_per_request: 1000\nmin_chunk_size: 1m\nmax_chunk_size: 4m\nmax_request_size: 100m\nmax_chunks_per_folder: 0\nmax_task_count: 100\nmax_tasks_per_node_per_request: 16\naio_max_pool_connections: 64\nmetadata_mem_cache_size: 128m\nmetadata_mem_cache_expire: 3600\nchunk_mem_cache_size: 128m\nchunk_mem_cache_expire: 3600\ndata_cache_size: 128m\ndata_cache_max_req_size: 128k\ndata_cache_expire_time: 3600\ndata_cache_page_size: 4m\ndata_cache_max_concurrent_read: 16\ntimeout: 30\npassword_file: /config/passwd.txt\ngroups_file: /config/groups.txt\nserver_name: Highly Scalable Data Service (HSDS)\ngreeting: Welcome to HSDS!\nabout: HSDS is a webservice for HDF data\ntop_level_domains: []\ncors_domain: "*"\nadmin_user: admin\nadmin_group: null\nopenid_provider: azure\nopenid_url: null\nopenid_audience: null\nopenid_claims: unique_name,appid,roles\nchaos_die: 0\nstandalone_app: false\nblosc_nthreads: 2\nhttp_compression: false\nhttp_max_url_length: 512\nk8s_app_label: hsds\nk8s_namespace: null\nrestart_policy: on-failure\ndomain_req_max_objects_limit: 500\n'
        tmp_dir = Path(tempfile.mkdtemp(prefix='tmp-hsds-'))
        config_file = tmp_dir / 'config.yml'
        config_file.write_text(config)
        passwd_file = tmp_dir / 'passwd.txt'
        passwd_file.write_text(f'admin:admin\n{os.environ['HS_USERNAME']}:{os.environ['HS_PASSWORD']}\n')
        log_file = str(tmp_dir / 'hsds.log')
        tmp_dir = str(tmp_dir)
        if sys.platform == 'darwin':
            socket_dir = '/tmp/hsds'
        else:
            socket_dir = tmp_dir
        try:
            hsds = HsdsApp(username=os.environ['HS_USERNAME'], password=os.environ['HS_PASSWORD'], password_file=str(passwd_file), log_level=os.getenv('LOG_LEVEL', 'DEBUG'), logfile=log_file, socket_dir=socket_dir, config_dir=tmp_dir, dn_count=2)
            hsds.run()
            is_up = hsds.ready
            if is_up:
                os.environ['HS_ENDPOINT'] = hsds.endpoint
                set_hsds_root()
        except Exception:
            is_up = False
        yield is_up
        hsds.stop()
        rmtree(tmp_dir, ignore_errors=True)
        rmtree(socket_dir, ignore_errors=True)
        rmtree(root_dir, ignore_errors=True)
    else:
        yield False