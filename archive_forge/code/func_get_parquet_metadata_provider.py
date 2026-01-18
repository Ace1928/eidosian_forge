from typing import List, Optional
from ray.data.datasource import (
from ray.data.datasource.image_datasource import _ImageFileMetadataProvider
def get_parquet_metadata_provider():
    return DefaultParquetMetadataProvider()