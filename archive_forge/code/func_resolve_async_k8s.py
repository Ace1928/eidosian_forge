from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_async_k8s(required: bool=True):
    """
    Ensures that `kubernetes_asyncio` is availableable
    """
    global _async_k8_available
    global AsyncClient, AsyncConfig, AsyncStream, AsyncWatch, AsyncUtils, AsyncType, AsyncWSClient
    if not _async_k8_available:
        resolve_missing('kubernetes_asyncio', required=required)
        import kubernetes_asyncio.client as AsyncClient
        import kubernetes_asyncio.config as AsyncConfig
        import kubernetes_asyncio.stream as AsyncStream
        import kubernetes_asyncio.watch as AsyncWatch
        import kubernetes_asyncio.utils as AsyncUtils
        import kubernetes_asyncio.client.models as AsyncType
        AsyncWSClient = AsyncStream.WsApiClient
        _async_k8_available = True