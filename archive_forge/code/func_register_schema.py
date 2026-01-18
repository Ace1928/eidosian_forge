from typing import List
import onnx.onnx_cpp2py_export.defs as C  # noqa: N812
from onnx import AttributeProto, FunctionProto
def register_schema(schema: OpSchema) -> None:
    """Register a user provided OpSchema.

    The function extends available operator set versions for the provided domain if necessary.

    Args:
        schema: The OpSchema to register.
    """
    version_map = C.schema_version_map()
    domain = schema.domain
    version = schema.since_version
    min_version, max_version = version_map.get(domain, (version, version))
    if domain not in version_map or not min_version <= version <= max_version:
        min_version = min(min_version, version)
        max_version = max(max_version, version)
        C.set_domain_to_version(schema.domain, min_version, max_version)
    C.register_schema(schema)