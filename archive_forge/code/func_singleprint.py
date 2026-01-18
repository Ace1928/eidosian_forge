from tensorflow.core.protobuf import fingerprint_pb2
from tensorflow.python.saved_model.pywrap_saved_model import fingerprinting as fingerprinting_pywrap
from tensorflow.python.util.tf_export import tf_export
def singleprint(self):
    """Canonical fingerprinting ID for a SavedModel.

    Uniquely identifies a SavedModel based on the regularized fingerprint
    attributes. (saved_model_checksum is sensitive to immaterial changes and
    thus non-deterministic.)

    Returns:
      The string concatenation of `graph_def_program_hash`,
      `signature_def_hash`, `saved_object_graph_hash`, and `checkpoint_hash`
      fingerprint attributes (separated by '/').

    Raises:
      ValueError: If the fingerprint fields cannot be used to construct the
      singleprint.
    """
    try:
        return fingerprinting_pywrap.Singleprint(self.graph_def_program_hash, self.signature_def_hash, self.saved_object_graph_hash, self.checkpoint_hash)
    except (TypeError, fingerprinting_pywrap.FingerprintException) as e:
        raise ValueError(f'Encounted invalid fingerprint values when constructing singleprint.graph_def_program_hash: {self.graph_def_program_hash}signature_def_hash: {self.signature_def_hash}saved_object_graph_hash: {self.saved_object_graph_hash}checkpoint_hash: {self.checkpoint_hash}{e}') from None