import pyarrow as pa
def record_batch(jvm_vector_schema_root):
    """
    Construct a (Python) RecordBatch from a JVM VectorSchemaRoot

    Parameters
    ----------
    jvm_vector_schema_root : org.apache.arrow.vector.VectorSchemaRoot

    Returns
    -------
    record_batch: pyarrow.RecordBatch
    """
    pa_schema = schema(jvm_vector_schema_root.getSchema())
    arrays = []
    for name in pa_schema.names:
        arrays.append(array(jvm_vector_schema_root.getVector(name)))
    return pa.RecordBatch.from_arrays(arrays, pa_schema.names, metadata=pa_schema.metadata)