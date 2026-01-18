from datetime import timedelta
import pyarrow.fs as fs
import pyarrow as pa
import pytest
@pytest.mark.skipif(encryption_unavailable, reason='Parquet Encryption is not currently enabled')
def test_dataset_encryption_decryption():
    table = create_sample_table()
    encryption_config = create_encryption_config()
    decryption_config = create_decryption_config()
    kms_connection_config = create_kms_connection_config()
    crypto_factory = pe.CryptoFactory(kms_factory)
    parquet_encryption_cfg = ds.ParquetEncryptionConfig(crypto_factory, kms_connection_config, encryption_config)
    parquet_decryption_cfg = ds.ParquetDecryptionConfig(crypto_factory, kms_connection_config, decryption_config)
    pformat = pa.dataset.ParquetFileFormat()
    write_options = pformat.make_write_options(encryption_config=parquet_encryption_cfg)
    mockfs = fs._MockFileSystem()
    mockfs.create_dir('/')
    ds.write_dataset(data=table, base_dir='sample_dataset', format=pformat, file_options=write_options, filesystem=mockfs)
    pformat = pa.dataset.ParquetFileFormat()
    with pytest.raises(IOError, match='no decryption'):
        ds.dataset('sample_dataset', format=pformat, filesystem=mockfs)
    pq_scan_opts = ds.ParquetFragmentScanOptions(decryption_config=parquet_decryption_cfg)
    pformat = pa.dataset.ParquetFileFormat(default_fragment_scan_options=pq_scan_opts)
    dataset = ds.dataset('sample_dataset', format=pformat, filesystem=mockfs)
    assert table.equals(dataset.to_table())